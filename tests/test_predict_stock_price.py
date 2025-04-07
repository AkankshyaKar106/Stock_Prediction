import unittest
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta
import pytz
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from final_model import predict_stock_price  # Replace with actual module


class TestPredictStockPrice(unittest.TestCase):

    @patch('final_model.st')
    @patch('final_model.fetch_bse_sensex_tickers')
    @patch('final_model.fetch_stock_data')
    @patch('final_model.process_ticker_data')
    @patch('final_model.load_lstm_model')
    @patch('final_model.load_rf_model')
    @patch('final_model.save_prediction')
    @patch('final_model.get_next_market_datetime')
    def test_predict_stock_price_next_hour(self, mock_market_dt, mock_save,
                                           mock_rf, mock_lstm, mock_process,
                                           mock_fetch_data, mock_fetch_tickers,
                                           mock_st):
        # Setup timezone
        ist_timezone = pytz.timezone('Asia/Kolkata')
        current_ist_time = datetime.now(ist_timezone)

        # Mock Streamlit inputs
        mock_st.selectbox.return_value = "NSE Nifty 500 (^NSEI)"
        mock_st.multiselect.return_value = ["RELIANCE.NS"]
        mock_st.radio.return_value = "Next Hour"
        mock_st.button.side_effect = [True]  # Simulate button click

        # Mock external functions
        mock_fetch_tickers.return_value = ["RELIANCE.NS"]
        mock_fetch_data.return_value = MagicMock(empty=False)
        mock_process.return_value = {
            "Ticker": "RELIANCE.NS",
            "Prediction Type": "Next Hour",
            "Prediction Created": current_ist_time,
            "Target Prediction Date": current_ist_time.date(),
            "Target Prediction Time": current_ist_time.time(),
            "Predicted Price": 2500.0,
            "Signal": "Buy",
            "Reason": "Uptrend",
            "Sentiment Score": 0.75,
        }
        mock_lstm.return_value = (MagicMock(), MagicMock())
        mock_rf.return_value = (MagicMock(), MagicMock(), MagicMock(),
                                MagicMock())
        mock_market_dt.return_value = (current_ist_time +
                                       timedelta(hours=1), "")

        predict_stock_price()

        # Assertions
        mock_fetch_tickers.assert_called_once()
        mock_fetch_data.assert_called()
        mock_save.assert_called()
        mock_process.assert_called()


if __name__ == '__main__':
    unittest.main()
