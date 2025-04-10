import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import pytz

from backend.final_model_utils import (
    load_lstm_model, load_rf_model, fetch_bse_sensex_tickers,
    fetch_nse_nifty_500_tickers,fetch_stock_data, get_next_market_datetime,
    process_ticker_data, create_predictions_table, save_prediction
)


def predict_stock_price():
    st.title("Stock Price Prediction")

    ist_timezone = pytz.timezone('Asia/Kolkata')
    current_ist_time = datetime.now(ist_timezone)

    indices = {
        "BSE Sensex (^BSESN)": lambda: fetch_bse_sensex_tickers(),
        "NSE Nifty 500 (^NSEI)": lambda: fetch_nse_nifty_500_tickers(),
    }

    selected_index = st.selectbox("Select an Index", list(indices.keys()))

    try:
        tickers = indices[selected_index]()
        if not tickers:
            st.error(f"No tickers found for {selected_index}")
            return
    except Exception as e:
        st.error(f"Error fetching tickers: {str(e)}")
        return

    selected_tickers = st.multiselect("Select Tickers", tickers)

    prediction_option = st.radio(
        "Predict for:", [
            "Next Hour", "Next Day", "Custom Time"])

    if prediction_option == "Custom Time":
        if 'custom_date' not in st.session_state:
            st.session_state.custom_date = current_ist_time.date()
        if 'custom_time' not in st.session_state:
            st.session_state.custom_time = current_ist_time.time()

        prediction_date = st.date_input(
            "Select Prediction Date",
            value=st.session_state.custom_date,
            min_value=current_ist_time.date(),
            key='date_input'
        )
        st.session_state.custom_date = prediction_date

        prediction_time = st.time_input(
            "Select Prediction Time",
            value=st.session_state.custom_time,
            key='time_input'
        )
        st.session_state.custom_time = prediction_time

        prediction_datetime = datetime.combine(
            prediction_date, prediction_time)
        prediction_datetime = ist_timezone.localize(prediction_datetime)

        if prediction_datetime < current_ist_time:
            st.warning(
                "Selected time is in the past. Please select a future time.")
            return

        prediction_type = "Custom Time"
    elif prediction_option == "Next Day":
        today = datetime.now()
        user_selected_date = st.date_input(
            "Select Prediction Date", value=today, min_value=today)

        try:
            prediction_datetime = datetime.combine(
                user_selected_date, datetime.min.time())
            date_str = user_selected_date.strftime('%Y-%m-%d')
            prediction_type = f"Prediction for {date_str}"
        except ValueError:
            print("Invalid date format. Using next day's date as default.")
            prediction_datetime = datetime.now() + timedelta(days=1)
            prediction_type = "Next Day (Default)"
    else:
        prediction_datetime = current_ist_time + timedelta(hours=1)
        prediction_type = "Next Hour"

    adjusted_datetime, market_message = get_next_market_datetime(
        prediction_datetime, prediction_type)

    if market_message:
        st.info(market_message)

    if st.button("Collect Data and Predict"):
        create_predictions_table()

        if 'prediction_attempt' not in st.session_state:
            st.session_state.prediction_attempt = 0

        st.session_state.prediction_attempt += 1

        attempt = st.session_state.prediction_attempt
        prediction_creation_key = f'prediction_created_at_{attempt}'

        st.session_state[prediction_creation_key] = datetime.now(ist_timezone)

        prediction_created_at = st.session_state[prediction_creation_key]

        if not selected_tickers:
            st.warning("Please select at least one ticker")
            return

        with st.spinner("Loading models..."):
            lstm_model, tokenizer = load_lstm_model()
            rf_model, scaler, target_scaler, sentiment_scaler = load_rf_model()

            if (lstm_model is None or rf_model is None or
                    tokenizer is None or scaler is None):
                st.error("Failed to load models or tokenizer")
                return

        all_prediction_data = []
        progress_bar = st.progress(0)

        for i, ticker in enumerate(selected_tickers):
            try:
                progress = (i + 1) / len(selected_tickers)
                progress_bar.progress(progress)

                status = st.empty()
                status.text(f"Processing {ticker}...")

                stock_data = fetch_stock_data(ticker, adjusted_datetime)

                if stock_data is None or stock_data.empty:
                    st.error(f"No data found for {ticker}")
                    continue

                with st.spinner(f"Computing indicators for {ticker}..."):
                    prediction_data = process_ticker_data(
                        ticker,
                        adjusted_datetime,
                        prediction_created_at,
                        stock_data,
                        lstm_model,
                        tokenizer,
                        rf_model,
                        scaler,
                        sentiment_scaler,
                        target_scaler,
                        prediction_type
                    )

                    if prediction_data:
                        all_prediction_data.append(prediction_data)
                        st.write(f"Results for {ticker}:")
                        try:
                            save_prediction(prediction_data)
                            st.write(
                                f"Successfully saved prediction for {ticker}")
                        except Exception as e:
                            st.error(
                                (f"Error saving prediction for {ticker}:"
                                 f"{str(e)}"))
                            st.write(f"Results for {ticker}:")
                            st.write(prediction_data)
                status.empty()

            except Exception as e:
                st.error(f"Error processing {ticker}: {str(e)}")

        progress_bar.empty()

        if all_prediction_data:
            st.success("Analysis complete!")
            prediction_df = pd.DataFrame(all_prediction_data)

            column_order = [
                "Ticker",
                "Prediction Type",
                "Prediction Created",
                "Target Prediction Date",
                "Target Prediction Time",
                "Predicted Price",
                "Signal",
                "Reason",
                "Sentiment Score"
            ]
            prediction_df = prediction_df[column_order]

            st.write("Summary of Predictions:")
            st.dataframe(prediction_df)

            datetime_fmt = '%Y-%m-%d %H:%M:%S IST'
            datetime_str = prediction_created_at.strftime(datetime_fmt)
            st.write(f"Predictions generated at: {datetime_str}")


if __name__ == "__main__":
    predict_stock_price()
