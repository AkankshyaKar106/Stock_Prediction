import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf
import pytz


def get_target_price(ticker, target_date, target_time):
    try:
        target_datetime = datetime.strptime(
            f"{target_date} {target_time}", '%Y-%m-%d %H:%M:%S')

        ist = pytz.timezone('Asia/Kolkata')
        target_datetime = ist.localize(target_datetime)
        start_date = target_datetime.date()
        end_date = start_date + timedelta(days=1)

        for suffix in ['.NS', '.BO', '']:
            try:
                stock = yf.Ticker(ticker + suffix)
                hist = stock.history(
                    start=start_date, end=end_date, interval='1m')

                if not hist.empty:
                    hist_times = pd.to_datetime(hist.index).tz_convert(
                        'Asia/Kolkata')
                    closest_time_idx = (
                        abs(hist_times - target_datetime)).argmin()
                    closest_price = hist['Close'][closest_time_idx]
                    actual_time = hist_times[closest_time_idx]

                    return closest_price, actual_time

            except Exception as e:
                print(
                    f"Error fetching historical price for {ticker + suffix}: "
                    f"{str(e)}"
                )
                continue

        return None, None

    except Exception as e:
        print(f"Error in get_target_price: {str(e)}")
        return None, None


def calculate_prediction_accuracy(predicted_price, actual_price):
    if predicted_price is None or actual_price is None:
        return None, None

    absolute_diff = abs(predicted_price - actual_price)
    percentage_diff = (absolute_diff / actual_price) * 100

    return absolute_diff, percentage_diff
