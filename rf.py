import pandas as pd
import numpy as np
import talib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
import pickle


def load_stock_data(stock_file):
    stock_data = pd.read_csv(stock_file, index_col=[0, 1], low_memory=False)
    stock_data.reset_index(inplace=True)
    stock_data["Previous_Close"] = stock_data["Close"].shift(1)
    stock_data.ffill(inplace=True)
    return stock_data


def load_sentiment_data(sentiment_file):
    sentiment_data = pd.read_csv(sentiment_file)
    sentiment_agg = sentiment_data.groupby(
        ["Ticker"])["Sentiment_Score"].mean().reset_index()
    return sentiment_agg


def merge_data(stock_data, sentiment_data):
    merged_data = pd.merge(stock_data, sentiment_data, on="Ticker", how="left")
    merged_data["Sentiment_Score"].fillna(0, inplace=True)
    return merged_data


def prepare_features(data, sentiment_scaler=None):
    df = data.copy()

    close_array = df['Close'].values
    high_array = df['High'].values
    low_array = df['Low'].values

    df['SMA_20'] = talib.SMA(df['Close'].values, timeperiod=20)
    df['SMA_50'] = talib.SMA(df['Close'].values, timeperiod=50)
    df['RSI'] = talib.RSI(df['Close'].values, timeperiod=14)
    df['MACD'], df['MACD_Signal'], _ = talib.MACD(df['Close'].values)
    df['BB_Upper'], df['BB_Middle'], df['BB_Lower'] = talib.BBANDS(
        df['Close'].values)
    df['Price_Change'] = df['Close'].pct_change()

    if 'Volume' in df.columns:
        df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')
        df['Volume_Change'] = df['Volume'].pct_change()
        df['Volume_MA_5'] = df['Volume'].rolling(window=5).mean()

    df['ATR'] = talib.ATR(high_array, low_array, close_array, timeperiod=14)
    df['ROC'] = talib.ROC(close_array, timeperiod=10)
    df['MOM'] = talib.MOM(close_array, timeperiod=10)

    df = df.ffill().bfill()

    if sentiment_scaler is not None:
        df['Sentiment_Score'] = sentiment_scaler.transform(
            df[['Sentiment_Score']])

    return df


def train_model(data, features, target, model_path, scaler_path,
                target_scaler_path, sentiment_scaler_path):
    x = data[features]
    y = data[target]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42)

    scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()
    sentiment_scaler = MinMaxScaler()

    x_train_scaled = scaler.fit_transform(
        x_train.drop(columns='Sentiment_Score'))
    x_test_scaled = scaler.transform(x_test.drop(columns='Sentiment_Score'))

    x_train_sentiment_scaled = sentiment_scaler.fit_transform(x_train[[
        'Sentiment_Score']])
    x_test_sentiment_scaled = sentiment_scaler.transform(x_test[[
        'Sentiment_Score']])

    x_train_scaled = np.hstack([x_train_scaled, x_train_sentiment_scaled])
    x_test_scaled = np.hstack([x_test_scaled, x_test_sentiment_scaled])

    y_train_scaled = target_scaler.fit_transform(y_train.values.reshape(-1, 1))

    rf_model = RandomForestRegressor(n_estimators=100, max_depth=20,
                                     random_state=42)
    rf_model.fit(x_train_scaled, y_train_scaled.ravel())

    with open(model_path, "wb") as f:
        pickle.dump(rf_model, f)
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    with open(target_scaler_path, "wb") as f:
        pickle.dump(target_scaler, f)
    with open(sentiment_scaler_path, "wb") as f:
        pickle.dump(sentiment_scaler, f)

    return rf_model, scaler, target_scaler, sentiment_scaler


def predict_next_day(
        rf_model, latest_data, features, scaler,
        ticker, target_scaler, sentiment_scaler):
    if latest_data.empty:
        return pd.DataFrame()

    latest_data_df = latest_data[features].iloc[-1].fillna(0).to_frame().T

    latest_data_scaled = scaler.transform(
        latest_data_df.drop(columns=['Sentiment_Score']))
    latest_data_sentiment_scaled = sentiment_scaler.transform(
        latest_data_df[['Sentiment_Score']])
    latest_data_scaled = np.hstack([
        latest_data_scaled, latest_data_sentiment_scaled])

    predicted_price = rf_model.predict(latest_data_scaled)[0]
    predicted_price = target_scaler.inverse_transform(
        [[predicted_price]])[0][0]

    today_price = (latest_data["Previous_Close"].iloc[-1]
                   if "Previous_Close" in latest_data.columns else None)
    signal = ("BUY" if today_price and predicted_price > today_price
              else "SELL" if today_price and predicted_price < today_price
              else "HOLD")
    reason = f"Predicted price ({predicted_price:.2f})"
    f"{'higher' if signal == 'BUY' else 'lower'} than today's price"
    f"({today_price:.2f})" if today_price else "Insufficient data"
    prediction_date = (datetime.today() + timedelta(days=1)).strftime(
        '%Y-%m-%d')

    return pd.DataFrame([{
        "Ticker": ticker,
        "Prediction Date": prediction_date,
        "Predicted Price": round(predicted_price, 2),
        "Signal": signal,
        "Reason": reason
    }])


def predict_next_hour(
        rf_model, latest_data, features, scaler, ticker,
        target_scaler, sentiment_scaler):
    if latest_data.empty:
        return pd.DataFrame()

    latest_data_df = latest_data[features].iloc[-1].fillna(0).to_frame().T
    latest_data_scaled = scaler.transform(latest_data_df.drop(
        columns=['Sentiment_Score']))
    latest_data_sentiment_scaled = sentiment_scaler.transform(
        latest_data_df[['Sentiment_Score']])
    latest_data_scaled = np.hstack(
        [latest_data_scaled, latest_data_sentiment_scaled])

    predicted_price = rf_model.predict(latest_data_scaled)[0]
    predicted_price = target_scaler.inverse_transform(
        [[predicted_price]])[0][0]

    current_price = latest_data["Close"].iloc[-1]
    signal = ("BUY" if predicted_price > current_price else "SELL"
              if predicted_price < current_price else "HOLD")
    reason = f"Predicted next-hour price ({predicted_price:.2f})"
    f"{'higher' if signal == 'BUY' else 'lower'} than current price"
    f"({current_price:.2f})"
    prediction_time = (datetime.now() + timedelta(hours=1)).strftime(
        '%Y-%m-%d %H:%M:%S')

    return pd.DataFrame([{
        "Ticker": ticker,
        "Prediction Time": prediction_time,
        "Predicted Price": round(predicted_price, 2),
        "Signal": signal,
        "Reason": reason
    }])


if __name__ == "__main__":
    stock_csv = "D:/Qwegle/Qwegle/Algorithmic_Trading/stock_data.csv"
    sentiment_csv = "D:/Qwegle/Qwegle/Algorithmic_Trading/sentiment_score.csv"
    model_path = "random_forest.pkl"
    scaler_path = "scaler1.pkl"
    target_scaler_path = "target_scaler1.pkl"
    sentiment_scaler_path = "sentiment_scaler1.pkl"
    selected_ticker = "ADANIENT.NS"

    stock_data = load_stock_data(stock_csv)
    sentiment_data = load_sentiment_data(sentiment_csv)
    data = merge_data(stock_data, sentiment_data)
    data = prepare_features(data)
    features = [
        "Previous_Close", "Open", "High", "Low", "Volume", "SMA_20", "SMA_50",
        "RSI", "MACD", "BB_Middle", "Sentiment_Score", "ATR", "ROC", "MOM"]
    target = "Close"

    rf_model, scaler, target_scaler, sentiment_scaler = train_model(
        data, features, target, model_path, scaler_path,
        target_scaler_path, sentiment_scaler_path)

    selected_stock_data = data[data['Ticker'] == selected_ticker]

    prediction_df = predict_next_day(
        rf_model, selected_stock_data, features, scaler, selected_ticker,
        target_scaler, sentiment_scaler)

    print(prediction_df)
