import pandas as pd
import sqlite3
import streamlit as st

from backend.compare_utils import (get_target_price,
                                   calculate_prediction_accuracy)


def analyze_predictions():
    st.title("Stock Price Prediction Analysis")
    conn = sqlite3.connect('stock_predictions.db')

    if st.button("Clear All Predictions"):
        cursor = conn.cursor()
        cursor.execute("DELETE FROM predictions")
        conn.commit()
        conn.close()
        st.success(
            "All previous predictions have been cleared from the database.")
        return

    if st.button("Compare Predictions with Target Prices"):
        predictions_df = pd.read_sql_query("SELECT * FROM predictions", conn)
        conn.close()

        if predictions_df.empty:
            st.warning("No predictions found in database")
            return

        predictions_df['Actual Price'] = None
        predictions_df['Price Difference'] = None
        predictions_df['Percentage Difference'] = None

        progress_bar = st.progress(0)
        status = st.empty()

        results_container = st.container()

        with results_container:
            st.subheader("Individual Prediction Results:")

            for idx, row in predictions_df.iterrows():
                progress = (idx + 1) / len(predictions_df)
                progress_bar.progress(progress)
                status.text(f"Processing {row['ticker']}...")

                try:
                    actual_price, actual_time = get_target_price(
                        row['ticker'], row['target_date'], row['target_time'])

                    if actual_price is not None:
                        abs_diff, pct_diff = calculate_prediction_accuracy(
                            row['predicted_price'], actual_price)

                        st.write(f"**Ticker: {row['ticker']}**")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.write(f"Target Date: {row['target_date']}")
                            st.write(f"Target Time: {row['target_time']}")
                        with col2:
                            st.write(
                                f"Predicted Price: "
                                f"₹{row['predicted_price']:.2f}")
                            st.write(
                                f"Price at Target Time: ₹{actual_price:.2f}")
                        with col3:
                            st.write(f"Difference: ₹{abs_diff:.2f}")
                            st.write(f"% Difference: {pct_diff:.2f}%")
                        st.markdown("---")

                        predictions_df.at[idx, 'Actual Price'] = actual_price
                        predictions_df.at[idx, 'Actual Time'] = actual_time
                        predictions_df.at[idx, 'Price Difference'] = abs_diff
                        predictions_df.at[
                            idx, 'Percentage Difference'] = pct_diff
                    else:
                        st.warning(
                            f"Could not fetch target time price for "
                            f"{row['ticker']}"
                        )
                except Exception as e:
                    st.error(f"Error processing {row['ticker']}: {str(e)}")

        progress_bar.empty()
        status.empty()

    conn.close()


if __name__ == "__main__":
    analyze_predictions()
