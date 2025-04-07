import unittest
from unittest.mock import patch, MagicMock
import pandas as pd

from your_module import analyze_predictions  # Replace with actual module

class TestAnalyzePredictions(unittest.TestCase):

    @patch('compare.st')
    @patch('compare.sqlite3.connect')
    @patch('compare.get_target_price')
    @patch('compare.calculate_prediction_accuracy')
    def test_compare_predictions(self, mock_accuracy, mock_get_price, mock_sql_connect, mock_st):
        # Setup fake DB response
        conn_mock = MagicMock()
        cursor_mock = MagicMock()
        mock_sql_connect.return_value = conn_mock
        conn_mock.cursor.return_value = cursor_mock

        # Simulate Streamlit input
        mock_st.button.side_effect = [False, True]  # Only second button (Compare Predictions) is clicked

        df = pd.DataFrame([{
            "ticker": "TCS.NS",
            "predicted_price": 3000.0,
            "target_date": "2025-04-08",
            "target_time": "10:30:00"
        }])
        pd.read_sql_query = MagicMock(return_value=df)

        mock_get_price.return_value = (3050.0, "10:30:00")
        mock_accuracy.return_value = (50.0, 1.66)

        analyze_predictions()

        mock_get_price.assert_called_once_with("TCS.NS", "2025-04-08", "10:30:00")
        mock_accuracy.assert_called_once_with(3000.0, 3050.0)

if __name__ == '__main__':
    unittest.main()
