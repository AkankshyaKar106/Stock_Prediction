import sqlite3
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from datetime import datetime, timedelta, date, time
import pytz
import uuid
import uvicorn

from final_model import (
    load_lstm_model, load_rf_model, fetch_stock_data,
    get_next_market_datetime, process_ticker_data)

# Import comparison functions
from compare import get_target_price, calculate_prediction_accuracy

app = FastAPI(title="Stock Prediction and Analysis API")

# Database Configuration
DATABASE_URL = 'stock_predictions.db'
ist_timezone = pytz.timezone("Asia/Kolkata")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic Models
class PredictionRequest(BaseModel):
    Indices: str
    Ticker: str
    Prediction_Type: str
    Prediction_Created: datetime = Field(
        ..., description="Format: 'YYYY-MM-DD HH:MM:SS'")
    Target_Prediction_Date: date = Field(...,
                                         description="Format: 'YYYY-MM-DD'")
    Target_Prediction_Time: time = Field(..., description="Format: 'HH:MM'")
    Predicted_Price: float
    Signal: str
    Reason: str
    Sentiment_Score: float


# Database Initialization
@app.on_event("startup")
def startup_event():

    global lstm_model, tokenizer, rf_model, scaler
    global sentiment_scaler, target_scaler

    try:
        # Database Initialization
        conn = sqlite3.connect(DATABASE_URL)
        cursor = conn.cursor()

        # Create predictions table if it does not exist
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                indices TEXT NOT NULL,
                ticker TEXT NOT NULL,
                prediction_type TEXT NOT NULL,
                prediction_created TEXT NOT NULL,
                target_date TEXT NOT NULL,
                target_time TEXT NOT NULL,
                predicted_price REAL NOT NULL,
                signal TEXT NOT NULL,
                reason TEXT NOT NULL,
                sentiment_score REAL NOT NULL,
                actual_price REAL,
                actual_time TEXT,
                price_difference REAL,
                percentage_difference REAL,
                status TEXT DEFAULT 'pending'
            )
        ''')

        conn.commit()
        conn.close()

        print("Database initialized successfully.")

    except Exception as e:
        print(f"Database initialization error: {e}")

    try:
        # Load ML Models
        lstm_model, tokenizer = load_lstm_model()
        rf_model, scaler, sentiment_scaler, target_scaler = load_rf_model()

        print("Models loaded successfully.")

    except Exception as e:
        print(f"Model loading error: {e}")


# Include the predict_stock_price_endpoint from the second file
@app.get("/predict_stock_price/")
def predict_stock_price_endpoint(
    index: str = Query(...,
                       enum=["BSE Sensex (^BSESN)", "NSE Nifty 500 (^NSEI)"]),
    tickers: list[str] = Query(...),
    prediction_option: str = Query(
        ..., enum=["Next Hour", "Next Day", "Custom Time"]),
    prediction_date: str = None,
    prediction_time: str = None
):
    try:

        current_ist_time = datetime.now(ist_timezone)

        if prediction_option == "Custom Time":
            if not prediction_date or not prediction_time:
                return {"error": "Custom time requires both date and time."}
            prediction_datetime = ist_timezone.localize(
                datetime.strptime(
                    f"{prediction_date} {prediction_time}", "%Y-%m-%d %H:%M"))
        elif prediction_option == "Next Day":
            prediction_datetime = current_ist_time + timedelta(days=1)
        else:
            prediction_datetime = current_ist_time + timedelta(hours=1)

        adjusted_datetime, market_message = get_next_market_datetime(
            prediction_datetime, prediction_option)

        batch_prediction_id = str(uuid.uuid4())
        all_prediction_data = []
        failed_tickers = []
        prediction_created_at = datetime.now(ist_timezone)

        # Database connection
        conn = sqlite3.connect(DATABASE_URL)
        cursor = conn.cursor()

        for ticker in tickers:
            try:

                stock_data = fetch_stock_data(
                    ticker, adjusted_datetime.strftime('%Y-%m-%d'))
                if stock_data is None or len(stock_data) == 0:
                    failed_tickers.append({
                        "ticker": ticker, "reason": "No stock data available"})
                    continue

                prediction_data = process_ticker_data(
                    ticker, adjusted_datetime, prediction_created_at,
                    stock_data, lstm_model, tokenizer, rf_model, scaler,
                    sentiment_scaler, target_scaler, prediction_option)

                if prediction_data:
                    cursor.execute('''
                        INSERT INTO predictions
                        (indices, ticker, prediction_type, prediction_created,
                         target_date, target_time, predicted_price,
                         signal, reason, sentiment_score)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        index, ticker, prediction_option,
                        prediction_created_at.strftime('%Y-%m-%d %H:%M:%S'),
                        prediction_data["Target Prediction Date"],
                        prediction_data["Target Prediction Time"].replace(
                            " IST", ""),
                        float(prediction_data["Predicted Price"]),
                        prediction_data["Signal"],
                        prediction_data["Reason"],
                        float(prediction_data["Sentiment Score"])
                    ))

                    conn.commit()

                    all_prediction_data.append(prediction_data)
                else:
                    failed_tickers.append({
                        "ticker": ticker, "reason": (
                            "Processing failed to generate prediction data")})

            except Exception as e:

                failed_tickers.append(
                    {"ticker": ticker, "reason": f"Error: {str(e)}"})

        conn.close()

        if not all_prediction_data:
            return {
                "message": "No predictions generated.",
                "failed_tickers": failed_tickers
            }

        response = {
            "batch_prediction_id": batch_prediction_id,
            "predictions": [
                {
                    "Ticker": pred["Ticker"],
                    "Target Prediction Date": pred["Target Prediction Date"],
                    "Target Prediction Time": pred["Target Prediction Time"],
                    "Predicted Price": float(pred["Predicted Price"]),
                    "Signal": pred["Signal"],
                    "Reason": pred["Reason"],
                    "Sentiment Score": float(pred["Sentiment Score"])
                    }
                for pred in all_prediction_data
                ]
            }

        return response
        if failed_tickers:
            response["failures"] = failed_tickers

        return response

    except Exception as e:
 
        return {"error": "Internal Server Error", "details": str(e)}


# Additional endpoints from the first file
@app.get("/predictions")
async def get_all_predictions():
    """Retrieve all predictions from the database"""
    try:
        conn = sqlite3.connect(DATABASE_URL)
        predictions_df = pd.read_sql_query("SELECT * FROM predictions", conn)
        conn.close()

        # ✅ Fill NaN values using forward and backward fill
        predictions_df = predictions_df.ffill().bfill().fillna(0)

        # ✅ Ensure accuracy is displayed correctly
        if "accuracy" in predictions_df.columns:
            predictions_df["accuracy"] = 100 - predictions_df["accuracy"]

        for idx, row in predictions_df.iterrows():
            if row["actual_price"] and row["predicted_price"]:
                abs_diff, pct_diff = calculate_prediction_accuracy(
                    row["predicted_price"], row["actual_price"])
                predictions_df.at[idx, "price_difference"] = abs_diff
                predictions_df.at[idx, "percentage_difference"] = pct_diff

        return {
            "batch_prediction_id": str(uuid.uuid4()),
            "predictions": predictions_df.to_dict(orient="records")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

        
@app.get("/predictions/analyze")
async def analyze_stock_predictions():
    """Analyze stock predictions by comparing them with actual prices"""
    print("🔍 Fetching predictions where actual_price is NULL or 0"
          "and target_time has passed...")

    try:
        conn = sqlite3.connect(DATABASE_URL)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        print("🔍 Fetching updated predictions after analysis...")

        # ✅ Fix: Handle NULL values and ensure correct time comparison
        query = '''
            SELECT id, ticker, prediction_type, prediction_created,
            target_date, target_time, predicted_price, signal, reason,
            sentiment_score, actual_price, accuracy, status, indices
            FROM predictions
            WHERE (actual_price IS NULL OR actual_price = 0)
            -- Handle NULL values
            AND STRFTIME('%Y-%m-%d %H:%M:%S', target_date || ' ' ||
                SUBSTR(target_time || ':00', 1, 8)) < STRFTIME(
                    '%Y-%m-%d %H:%M:%S', 'now', 'localtime');
        '''
        print("SQL Query:\n", query)

        cursor.execute(query)
        predictions = cursor.fetchall()
        print(f"✅ Found {len(predictions)} pending predictions.")

        if not predictions:
            return {"message": "No predictions to analyze"}

        results = []

        for pred in predictions:
            pred_dict = dict(pred)  # ✅ Convert row to dictionary

            pred_id = pred_dict["id"]
            ticker = pred_dict["ticker"]
            target_date = pred_dict["target_date"]
            target_time = pred_dict["target_time"]
            predicted_price = pred_dict["predicted_price"]

            # ✅ Ensure time format is always HH:MM:SS
            formatted_time = target_time if len(
                target_time) == 8 else f"{target_time}:00"

            print(f"📌 Processing {ticker} | Target: {target_date}"
                  f"{formatted_time}")

            # Fetch actual price with corrected format
            actual_price, actual_time = get_target_price(
                ticker, target_date, formatted_time)

            if actual_price is not None:
                abs_diff, pct_diff = calculate_prediction_accuracy(
                    predicted_price, actual_price)

                print(
                    f"✅ Actual Price: {actual_price} at {actual_time}")
                print(f"📊 Price Difference: {abs_diff} |"
                      f" % Difference: {pct_diff}%")

                cursor.execute('''
                    UPDATE predictions
                    SET actual_price = ?,
                        price_difference = ?,
                        percentage_difference = ?,
                        accuracy = ?,
                        status = 'completed'
                    WHERE id = ?
                ''', (
                    actual_price, abs_diff, pct_diff, 100 - pct_diff, pred_id))

                conn.commit()

                # ✅ Add updated values to response
                pred_dict.update({
                    "actual_price": actual_price,
                    "price_difference": abs_diff,
                    "percentage_difference": pct_diff,
                    "accuracy": 100 - pct_diff,
                    "status": "completed"
                })

                results.append(pred_dict)

            else:
                print(f"⚠️ Could not fetch actual price for {ticker}")

        # ✅ **Re-fetch updated predictions**
        print("🔄 Fetching predictions after updates...")
        cursor.execute('''
            SELECT id, indices, ticker, target_date, target_time,
            predicted_price, actual_price, price_difference,
            percentage_difference, accuracy, status
            FROM predictions
            WHERE actual_price IS NOT NULL;
        ''')

        updated_predictions = cursor.fetchall()

        conn.close()
        print(f"📊 Final Updated Predictions: {updated_predictions}")

        return {"results": updated_predictions}

    except Exception as e:
        print(f"❌ Error in analyze_stock_predictions: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/predictions/clear")
async def clear_predictions():
    """Clear all predictions from the database"""
    try:
        conn = sqlite3.connect(DATABASE_URL)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM predictions")
        conn.commit()
        conn.close()

        return {"message": "All predictions cleared successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8002, reload=True)
