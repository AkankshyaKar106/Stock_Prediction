This project predicts stock prices using a hybrid approach that combines sentiment analysis powered by an LSTM model and stock data processed through a Random Forest regression model. It features a user-friendly **Streamlit web app** for interaction and a **FastAPI backend** for API-based prediction. Model evaluation and CI/CD integration are also included.

---

## 🧠 Key Features

- 💡 Predict stock prices using sentiment-enhanced models
- 📰 Sentiment analysis of financial news using LSTM
- 🧮 Random Forest regression for price prediction
- 🌐 Streamlit app for interactive use
- ⚙️ FastAPI backend for programmatic access
- 📊 Compare actual vs predicted stock prices
- 🔄 GitHub Actions workflow for CI/CD

## 📁 Project Structure

. ├── .github/workflows/ # CI/CD config 
  ├── dev-branch-checks.yml 
  ├── backend/ # (Optional) backend modules 
  ├── tests/ # Unit tests   
  ├── compare.py # Compares actual and predicted prices
  ├── final_model.py # Streamlit web app 
  ├── main.py # FastAPI backend 
  ├── lstm_model.py # Sentiment score calculator 
  ├── lstm.h5 # Trained LSTM model 
  ├── tokenizer1.pkl # Tokenizer for LSTM 
  ├── scaler1.pkl # Scaler for features 
  ├── sentiment_scaler1.pkl # Scaler for sentiment scores 
  ├── target_scaler1.pkl # Scaler for target values 
  ├── rf.py # Random Forest model logic 
  ├── requirements.txt # Python dependencies 
  ├── setup.cfg # Project configuration 
  ├── readme.MD # README file

---

## ⚙️ Setup Instructions

1. Clone the repository

```bash
git clone https://github.com/AkankshyaKar106/Stock_Prediction
cd Stock_Prediction

2. Install dependencies

pip install -r requirements.txt

3. Ensure the following model files are present in the root directory:

- lstm.h5
- tokenizer1.pkl
- scaler1.pkl
- sentiment_scaler1.pkl
- target_scaler1.pkl

4. Run the Streamlit App (Frontend)

- Launch the interactive web interface for predictions:
streamlit run final_model.py

- Launch the interactive web interface for predictions:
streamlit run compare.py

5. Run the FastAPI Server (Backend)

Start the API server:
uvicorn main:app --reload --port 8002
Once running, visit http://127.0.0.1:8002/docs to explore and test API endpoints via Swagger UI.
